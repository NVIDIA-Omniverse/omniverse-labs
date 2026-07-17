---
name: extend-ovrtx-materialx-openpbr
description: Extend and test Blender material graph conversion to MaterialX OpenPBR in the OVRTX Blender add-on. Use when adding supported shader nodes, sockets, texture operations, UV handling, material bindings, fallback policy, conversion diagnostics, or OpenPBR parity.
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
    - materialx
    - materials
---
# Extend OVRTX MaterialX/OpenPBR conversion

## When to Use

Use when adding supported shader nodes, sockets, texture operations, UV handling, material bindings, fallback policy, conversion diagnostics, or OpenPBR parity.

## Boundary

Work in the distributed source checkout and use documented runtime interfaces.
The add-on owns Blender graph discovery, conversion policy, texture materialization,
USD overlay generation, binding identity, caching, and diagnostics. Do not solve
unsupported graph conversion by patching the runtime or mutating the source
material.

## Code map

- `addon/ovrtx_blender_example/materialx_openpbr_conversion.py`
  - `scene_layer_from_materials`: conversion entry and fallback policy.
  - `_scene_overlay_from_materials`, `_resolve_binding`, `_classify_material`:
    selection, USD binding resolution, supported/unsupported classification.
  - `_active_material_surface`, `_surface_graph_nodes`, `_node_diagnostics`,
    `_blocking_reasons`: active graph discovery and fail-closed diagnostics.
  - `_openpbr_values_from_surface` and `_principled_openpbr_values`: surface and
    socket-to-OpenPBR value conversion.
  - `_texture_record_from_socket`, `_post_image_op_record`,
    `_normal_texture_record`, `_linked_texture_node`: texture graph extraction.
  - `_material_block_lines`, `_texture_shader_lines`, `_input_connect_line`,
    `_binding_tree_lines`: deterministic USDA emission and binding.
- `addon/ovrtx_blender_example/texture_materialization.py`:
  `materialized_image_path` and `texture_cache_directory` own disk/packed image
  resolution and content-addressed cache stability.
- `addon/ovrtx_blender_example/render_requests.py::MaterialPresentationLayer`:
  typed composition payload, authored properties, digest content, diagnostics.
- `addon/ovrtx_blender_example/blender_signal_translation.py::_material_scene_layer_from_scene`:
  selects MaterialX conversion, caches by source path/material identity, and
  preserves the exact-stage boundary.
- `addon/ovrtx_blender_example/usd_preview_emission_layer.py`: legacy final-render
  preview path; do not broaden it when the feature belongs to MaterialX/OpenPBR.

## Instructions

1. Define the Blender node/socket pattern, OpenPBR input, value type, units,
   colorspace, defaults, and unsupported cases. Decide whether the graph is a
   value extension, texture-chain extension, surface topology, or binding change.
2. Extend active-graph discovery and `_SUPPORTED_NODE_TYPES` only for nodes whose
   semantics are actually converted. Ensure disconnected nodes cannot block or
   influence output.
3. Add pure value conversion beside `_openpbr_values_from_surface` or the
   relevant `_apply_*_values` function. Clamp or transform only when the mapping
   specifies it; preserve omission semantics for inactive lobes.
4. For textures, extend `_TEXTURE_INPUTS` and extraction/emission together.
   Resolve packed and library-aware images through `materialized_image_path`.
   Mark scalar data raw and color data sRGB as appropriate. Model channel,
   normal, mapping, and post-image operations explicitly; never bake silently.
5. Extend `_material_block_lines`/`_texture_shader_lines` with deterministic
   identifiers and ordering. Include every output-affecting value in digest
   content so cache reuse cannot return stale overlays.
6. If binding behavior changes, update `_resolve_binding` and binding-tree
   emission without guessing a target from material name when identity is
   ambiguous. Invalid or absent targets must remain diagnostic, not misbound.
7. Preserve `allow_stock_fallback`: strict conversion returns
   `MaterialSceneConversionStatus.ERROR`; fallback records each skipped material
   and leaves its stock binding intact. Never claim a partial graph was converted.
8. Change `blender_signal_translation.py` only if conversion inputs or cache
   identity change. Keep exact-stage requests free of automatic replacement.

## Required tests

Extend `tests/test_materialx_openpbr_conversion.py` for:

- supported constants and boundary values;
- linked texture, packed image, colorspace, channel, mapping, and normal cases;
- disconnected/unsupported nodes and actionable `blocking_reasons`;
- deterministic layer text/digest and unique identifiers;
- strict error versus stock fallback;
- unbound, invalid, ambiguous, and multiple binding targets.

Extend `tests/test_texture_materialization.py` for any new image path/cache
behavior. Extend `tests/test_render_requests.py` when translation inputs, cache
keys, reuse/invalidation, or exact-stage behavior changes. Add a small Blender
fixture only when fake node graphs cannot establish the Blender API contract.

Run the focused tests, then the complete `tests` suite. With the supported
runtime, compare a generic textured multi-object scene between Blender reference
and OVRTX output. Check bindings, UV orientation, color/non-color handling,
opacity, normals, and lobe response; a successful render call alone is not a pass.

## Runtime escalation gate

Stay add-on-only when generated USD contains the intended MaterialX/OpenPBR
network and bindings but the add-on omitted or mistranslated Blender data.
Escalate only when a minimal hand-authored USD fixture using documented OpenPBR
nodes fails on a compatible installed runtime while the same runtime accepts its
documented baseline. Record fixture, runtime version, diagnostics, and output;
do not propose worker implementation changes from an unexplained visual mismatch.

## Handoff

Return changed files and symbols, supported and rejected graph patterns,
focused/full test results, generic visual comparison, cache/binding compatibility
notes, and `addon-only` or `runtime-capability-missing`.

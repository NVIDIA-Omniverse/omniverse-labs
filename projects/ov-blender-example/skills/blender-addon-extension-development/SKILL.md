---
name: blender-addon-extension-development
description: Route and implement source changes in the OVRTX Blender add-on. Use when adding Blender properties or UI, render settings, scene or USD conversion, interactive edits, render sequences, sensor products, viewport presentation, diagnostics, or tests.
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
    - addon-development
---
# Blender add-on extension development

## When to Use

Use when adding Blender properties or UI, render settings, scene or USD conversion, interactive edits, render sequences, sensor products, viewport presentation, diagnostics, or tests.

## Boundary

Work in the distributed source checkout and use its documented add-on and
runtime interfaces. Start with an add-on implementation and use a minimal
capability probe before reporting a missing runtime capability.

Use paths below relative to the source root. Preserve guarded `bpy` imports so pure
modules and tests remain importable outside Blender. Keep all worker calls on the
existing serialized session-owner/render thread.

## Instructions

Choose the narrowest route before editing:

| Feature | Start here | Continue through | Required tests |
| --- | --- | --- | --- |
| Blender property or panel | `addon/ovrtx_blender_example/properties.py` | `ui.py`, `engine.py`, typed request/composition module | focused property and panel tests such as `tests/test_rtpt_scene_properties.py` and `tests/test_rtpt_render_panel.py` |
| Render setting | `properties.py::RTPT_RENDER_SETTINGS` and `RtptSettingSpec` | `render_requests.py::RenderRequest`, `ovrtx_scene_composition.py`, `view_update_stream.py`, `runtime_scheduler.py` | `test_rtpt_scene_properties.py`, `test_rtpt_render_product_authoring.py`, `test_rtpt_live_change.py` |
| Material graph/OpenPBR | `materialx_openpbr_conversion.py::scene_layer_from_materials` | `texture_materialization.py`, `blender_signal_translation.py::_material_scene_layer_from_scene` | `test_materialx_openpbr_conversion.py`, `test_texture_materialization.py`, `test_render_requests.py` |
| Scene object/topology | `scene_generation.py::SceneGenerationOwner` | `_stock_export`, `_compile_add_on_opinions`, topology reconciliation, `topology_edit_fallback.py` | `test_scene_generation.py`, `test_scene_generation_contract.py`, `test_topology_edit_fallback.py` |
| Interactive value edit | `blender_signals.py` and `blender_signal_translation.py` | `blender_interactive_edit_builders.py`, `interactive_edit_planner.py`, `value_edit_conversion.py`, `interactive_edit_workflow.py`, `runtime_scheduler.py` | corresponding `test_blender_*`, `test_interactive_*`, `test_value_edit_conversion.py` |
| Session or sequence | `render_requests.py::RenderRequest` | `ovrtx_session.py`, `ovrtx_session_controller.py`, `runtime_scheduler.py`, `engine.py` | `test_ovrtx_session.py`, `test_ovrtx_session_controller.py`, `test_runtime_scheduler.py`, `test_final_render_on_rpc_thread.py` |
| Sensor/AOV result | `render_requests.py` and `ovrtx_runtime_client.py` | typed result model, scheduler publication, `engine.py` presentation | `test_render_requests.py`, `test_ovrtx_runtime_client.py`, scheduler and presentation tests |
| Viewport upload/display | `viewport_handoff.py` and `viewport_presentation.py` | `viewport_render_thread.py`, `engine.py` | `test_viewport_handoff.py`, `test_viewport_presentation.py`, `test_viewport_render_thread.py` |
| Preflight/diagnostics | `preflight.py` or the owning module's diagnostics model | `user_messages.py`, `ui.py` | `test_addon_preflight.py`, `test_user_messages.py`, focused diagnostics test |

For a specialized route, use the matching `extend-ovrtx-*` skill when present.

## Classify ownership before coding

Write down these decisions in the implementation plan:

1. Identify the Blender signal and the normalized add-on-owned value.
2. Classify it as durable USD topology, durable USD value, same-session runtime
   value, session identity, worker-startup configuration, or presentation-only.
3. Identify the typed request/result boundary. Do not pass arbitrary
   dictionaries across an existing dataclass boundary just to avoid modeling it.
4. Decide whether omission means `INHERIT`, whether an explicit value means
   `OVERRIDE`, and how reset returns to inherited behavior.
5. Identify the observable acceptance result: authored USD, effective setting,
   typed product, rendered pixels, or an explicit rejection diagnostic.

## Implementation

1. Add pure normalized types and conversion policy in a focused module. Reject
   unsupported input explicitly; do not silently approximate topology or units.
2. Add Blender properties and registration in `properties.py` and `__init__.py`
   only when the feature is user-configurable.
3. Carry the normalized value through `RenderRequest` or the appropriate edit
   intent/result type. Include it in identity/digests only when it changes the
   composed session.
4. Author durable USD through scene generation or composition. Route supported
   same-session values through the established edit/update stream. Replace the
   generation or session for topology and unsupported live writes.
5. Add UI after the typed path works. Display effective state and rejection or
   fallback reason, not only the requested value.
6. Add unit tests at each boundary and one included fixture/probe that observes
   the actual product with the installed runtime when available.

## Architectural invariants

- Keep Blender callbacks and UI code free of blocking runtime calls.
- Submit runtime work through `RuntimeScheduler`/`OvrtxSessionController`; do not
  instantiate another worker client in a property callback or operator.
- Keep current-scene generation separate from exact-stage validation. Exact-stage
  requests must not replace source materials, lights, or topology.
- Preserve `SceneGenerationOwner.replace`/`accept`/`reject`: a failed candidate
  must leave the last accepted generation usable.
- Keep material, lighting, color, and viewport presentation ownership singular.
  Do not compensate for a renderer problem by mutating source material values.
- Make cached keys depend on every input that changes output, and test both reuse
  and invalidation.
- Return typed results for non-image products; do not force point clouds or AOVs
  through an RGBA-only result model.
- Never report success from an accepted API call alone. Check the composed stage,
  effective state, product metadata, and visible/numeric output as applicable.

## Validation

Run the focused pure-Python tests first, then the complete `tests` suite.
For Blender-dependent behavior, run the documented included Blender fixture.
For runtime-dependent behavior, use the documented runtime interface and record
versions, request, response, diagnostics, and a caller-selected artifact path.

Test success, inheritance/reset, invalid input, missing optional capability,
cache invalidation, session replacement or reuse, cancellation/shutdown, and
plain-Python import behavior where relevant.

## Escalation gate

Report a runtime capability gap only when the smallest documented-interface
probe shows that the typed operation cannot be expressed, is rejected by a
compatible runtime, or cannot return enough metadata to distinguish success
from false success. Missing add-on routing, USD authoring, threading, UI,
conversion, caching, or diagnostics remains add-on work.

## Handoff

Return changed paths and symbols, focused and full-suite results, fixture
or visual evidence, compatibility notes, and exactly one boundary decision:
`addon-only` or `runtime-capability-missing` with the minimal probe.

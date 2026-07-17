---
name: extend-ovrtx-render-settings
description: Extend the OVRTX Blender add-on with renderer modes, RTPT controls, RenderProduct opinions, primvars, color presentation, diagnostics, and tests. Use when adding or changing renderer settings.
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
    - rendering
---
# Extend OVRTX render settings

## When to Use

Use when adding or changing renderer settings.

## Boundary

Work in a feature branch of the distributed source checkout and use documented
runtime interfaces. Use the capability probe below if the runtime cannot accept
or report the required typed setting.

## Code map

- `addon/ovrtx_blender_example/properties.py`: add the Blender property
  and its typed setting specification. Follow `RTPT_RENDER_SETTINGS` and
  `RtptSettingSpec` when the setting is a RenderProduct attribute.
- `addon/ovrtx_blender_example/ui.py`: draw the control and its effective
  state beside the existing render controls.
- `addon/ovrtx_blender_example/render_requests.py`: carry the normalized
  value in `RenderRequest`; include it in `snapshot_key` only when it changes
  session identity.
- `addon/ovrtx_blender_example/ovrtx_scene_composition.py`: author typed
  USDA in `_generated_presentation_body_lines` and include startup-only values
  in `_composition_digest`.
- `addon/ovrtx_blender_example/engine.py`: translate Blender state in
  `build_request_from_scene` and route live setting changes through
  `submit_render_setting_change_to_active_viewports`.
- `addon/ovrtx_blender_example/view_update_stream.py`: add exact
  attribute/type conversion only for settings supported by live writes.
- `addon/ovrtx_blender_example/runtime_scheduler.py`: preserve rejection
  diagnostics and request a session re-key when a live write is unsupported.
- `addon/ovrtx_blender_example/ovrtx_session_controller.py`: put
  worker-startup configuration or restart fallback on the serialized owner
  thread.

## Instructions

1. Start from current main branch and inventory the setting's exact documented
   token, USD type, valid range, default, ownership, and whether it applies at
   session creation or through a live write.
2. Add one source of truth mapping the Blender property to its RenderProduct or
   worker-startup representation. Expose `inherit` where omission preserves a
   source opinion.
3. Route the value through properties, UI, request identity, composition, live
   update or restart fallback, effective-state diagnostics, and reset behavior.
4. Keep color ownership singular. Do not compensate for renderer presentation
   with material mutations or a second transfer function.
5. Test property metadata, UI ordering, typed USDA, digest participation,
   live-write success and rejection, restart fallback, and round-trip effective
   state.
6. Run one small animated RTPT fixture and compare inherited, changed, and reset
   states. Preserve images, effective settings, runtime identity, and logs.

## Required tests

Add or extend `tests/test_rtpt_scene_properties.py`,
`test_rtpt_render_panel.py`, `test_rtpt_render_product_authoring.py`,
`test_rtpt_live_change.py`, `test_rtpt_live_write_fallback.py`, and
`test_rtpt_quality_diagnostics.py`. If the setting is not RTPT-specific, create
parallel narrowly named tests rather than putting unrelated behavior under an
RTPT filename.

## Client/server escalation gate

Use a minimal direct documented-interface fixture with the correct RenderProduct. A
client/server change is justified only if the official API cannot express the
typed value, the worker rejects a documented supported token, or effective
state cannot be observed sufficiently to prevent false success. Record the
request, response, versions, and smallest fixture. UI absence, stale add-on
composition, or missing fallback logic is add-on work, not worker work.

## Handoff

Return a focused commit, tests, visual A/B evidence, compatibility notes,
and an explicit `addon-only` or `client/server-escalation-required` decision.

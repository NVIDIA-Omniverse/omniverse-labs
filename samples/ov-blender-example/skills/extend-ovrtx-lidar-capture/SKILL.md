---
name: extend-ovrtx-lidar-capture
description: Extend the OVRTX Blender add-on with native LiDAR sensor authoring, runtime capability discovery, raw-return capture, coordinate conversion, accumulation, viewport presentation, and evidence. Use when adding LiDAR support rather than producing a synthetic point overlay.
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
    - sensors
---
# Extend OVRTX LiDAR capture

## When to Use

Use when adding LiDAR support rather than producing a synthetic point overlay.

## Boundary

Start in a feature branch of the distributed source checkout and use documented
sensor schemas and runtime interfaces. A visualization made from mesh samples
or a depth image is not native LiDAR support.

## Code map

- `addon/ovrtx_blender_example/authoring_properties.py`: define sensor
  profile, pose, range, rate, and return-policy properties when they belong to
  authored scene data; use `properties.py` for render-session controls.
- `addon/ovrtx_blender_example/blender_signal_translation.py`: translate
  Blender sensor state into immutable request values.
- `addon/ovrtx_blender_example/render_requests.py`: extend
  `RenderRequest.sensor_paths`/`selected_sensor_paths` with a typed product
  selection structure if LiDAR needs fields beyond a render-variable name.
- `addon/ovrtx_blender_example/ovrtx_scene_composition.py`: extend
  `_generated_presentation_body_lines` or add a dedicated sensor contribution
  that authors typed ancestors, sensor prim, product, and requested variables.
- `addon/ovrtx_blender_example/ovrtx_runtime_client.py`: use documented bindings
  and add add-on-side request construction and
  result decoding beside `_render_var_paths`, `render_result`, and
  `render_result_from_native`. Do not force point arrays through `RenderResult`;
  introduce a typed `SensorResult`.
- `addon/ovrtx_blender_example/ovrtx_session.py`: include immutable
  profile/shape fields in `reuse_decision`; allow pose/time changes to reuse a
  session when the installed API supports them.
- Add `lidar_capture.py` for coordinate conversion, raw persistence,
  accumulation, and derived viewport data. Keep these out of the runtime
  adapter.
- `addon/ovrtx_blender_example/ui.py`: expose capability, profile,
  capture, and raw/derived status without presenting synthetic points as native.

## Instructions

1. Probe supported sensor profiles, output fields, array shapes, coordinate
   convention, timing, immutable session fields, and capability errors.
2. Add Blender sensor properties and deterministic USD authoring with explicit
   profile, pose, frame rate, range, return policy, and typed ancestry.
3. Capture and retain raw points, return identity, timestamps, pose, range,
   intensity and labels when available before any filtering.
4. Convert to world space, validate an asymmetric isolated target, then add
   camera projection, first-surface visibility, decimation, accumulation, and
   viewport presentation as derived layers.
5. Test authoring and array parsing with fixtures. Run one short scan of a
   representative multi-object scene and validate finite values, bounds,
   pose/time changes, coverage, raw hashes, and truthful raw-versus-derived
   labeling.

## Required tests

Add `tests/test_lidar_authoring.py`, `test_lidar_capture.py`, and
`test_lidar_session_reuse.py`. Extend `test_ovrtx_runtime_client.py` with fake
installed bindings that return exact point arrays and capability failures;
extend `test_render_requests.py` and `test_generated_presentation_defs.py` for
request normalization and typed USD. Test handedness, row/column convention,
finite values, immutable raw arrays, pose/time identity, and unsupported fields.

## Client/server escalation gate

Run the smallest direct-client sensor fixture. Escalate only if the official
runtime exposes no native PointCloud/LiDAR product, omits required raw fields,
returns an unusable schema, or cannot advance/capture a supported sensor.
Missing Blender controls, authoring, transforms, accumulation, or visualization
are add-on responsibilities. Preserve the capability response before proposing
protocol, client, or worker changes.

## Handoff

Return the focused commit, authoring/parser tests, raw arrays, alignment fixture,
review images, capture report, and the boundary decision.

---
name: ovrtx-render-settings
description: Configure OVRTX renderer-owned settings for a Blender scene while preserving source USD opinions and making overrides inspectable. Use for OVRTX render mode, samples, backgrounds, exposure, light visibility, tone mapping, compositing, depth of field, ray visibility, or renderer primvars.
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
# OVRTX Render Settings

Configure renderer-owned settings for a Blender scene while preserving source
USD opinions and making every override inspectable.

## When to Use

Use for OVRTX render mode, samples, backgrounds, exposure, light visibility, tone mapping, compositing, depth of field, ray visibility, or renderer primvars.

## Boundary

- Use the installed OVRTX runtime through documented add-on interfaces.
- Keep source scene data authoritative. Apply settings through the add-on's
  supported UI/API or temporary composition layer, not hand-authored transport
  USD unless a diagnostic fixture is explicitly requested.
- Record add-on/runtime versions, active camera, render product, resolution, and
  effective settings in the output report.

## Instructions

1. Inspect the current scene and confirm the active OVRTX render product/camera.
   If the add-on lacks a setting, report that gap instead of inventing a property.
2. For every requested setting, show the inherited/source value and choose either
   `INHERIT` (author no opinion) or `OVERRIDE` (author one typed opinion).
3. Group controls by intent: render mode, background, sampling/light paths,
   lighting visibility, camera exposure, film/compositing, color/tone map, and
   per-prim camera/shadow/reflection visibility.
4. Validate tokens and value types against the installed OVRTX schema/runtime.
   Do not assume legacy spellings such as `RealTimePathTracing` or
   `omni:rtx:background:color`; use the current documented contract.
5. Apply only changed overrides, trigger the documented render/refinement path,
   and wait for a completed frame. A successful property write is not visual proof.
6. Capture a label-free image plus the effective-setting report. Pair with a
   parity/image-review skill for renderer comparisons.
7. On reset, remove the override and verify that the source/default value returns.

## Safety and diagnosis

- Do not use material edits to compensate for exposure, tone-map, or light-unit
  mismatches.
- Keep viewport refinement controls separate from final samples-per-pixel.
- If a setting is accepted by Blender but absent from the resolved OVRTX stage,
  classify it as unsupported and stop claiming that it was applied.
- Distinguish an unavailable runtime, unsupported schema token,
  composition-layer failure, and visual mismatch in the report.

## Handoff

Return scene/source identity, effective settings, explicit overrides,
render-product and camera paths, output image, logs, and unsupported or
runtime-dependent controls. Keep generated artifacts in a caller-provided
cache/output directory.

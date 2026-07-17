---
name: ovrtx-color-management
description: Keep Blender, OVRTX, and review-tool color handling explicit and single-applied. Use when choosing Scene Linear HDR versus LDR display passthrough, setting view transform/look/exposure/gamma, diagnosing washed-out or dark renders, or comparing OVRTX with Blender references.
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
    - color-management
---
# OVRTX color management

Color correctness is a data-contract problem before it is a look-development
problem. Treat the installed add-on/runtime as authoritative for available
frame formats and keep source, native, and display derivatives separate.

## When to Use

Use when choosing Scene Linear HDR versus LDR display passthrough, setting view transform/look/exposure/gamma, diagnosing washed-out or dark renders, or comparing OVRTX with Blender references.

## Choose one presentation mode

- `scene_linear_hdr`: request `HdrColor`/RGBA16F when available. The consumer
  owns the display transform; record the OCIO/display device and transform used
  for every review derivative.
- `ldr_rgba8_display_passthrough`: request `LdrColor`/RGBA8 when the OVRTX
  render product already applies its display transform. Do not apply Blender's
  view transform again.
- `ocio_baked_display`: treat as unavailable unless the add-on/runtime reports
  an explicit implementation. Never silently substitute it.

The example's documented color-presentation diagnostics expose the requested and
active mode, frame format, RenderVar, conversion, display-transform owner, and
Blender view settings. Include those fields in the capture report.

## Instructions

1. Freeze the scene camera, resolution, world/light settings, render engine,
   view transform, look, exposure, gamma, display device, and output format.
2. Render a neutral gray/chrome or checker control before tuning materials.
   Capture the same frame as HDR and/or LDR when supported and retain raw data.
3. Apply a view/exposure change once, render again, and verify the pixel data or
   measured luma changes in the expected direction. Never diagnose a double
   transform from a screenshot alone.
4. Compare Blender/Cycles and OVRTX only after camera, geometry, light units,
   world, and material identity are documented. Label comparisons as look
   review, not pixel parity, when cameras or transforms differ.
5. If output is black, clipped, flat, or over-bright, diagnose in order:
   runtime/readback, product/RenderVar, frame format, transform ownership,
   exposure/tone map, then lights/materials. Do not recolor albedo to hide a
   display error.

## Evidence requirements

For every image write a sidecar containing source/runtime identity, requested
and active color mode, frame format, RenderVar, transform owner, Blender view
settings, and any conversion code/version. Keep source-linear EXR/NPY and
display PNG separate. State whether a derivative is native, consumer-converted,
or postprocessed review output.

If the requested mode or RenderVar is unsupported, report `unavailable` with the
runtime diagnostic. Do not claim color-managed OVRTX output from an EEVEE or
Cycles fallback.

---
name: ovrtx-render-products-and-aovs
description: Configure and capture OVRTX render products and AOVs from a Blender scene, keeping native buffers, render-variable identity, and review images separate. Use for beauty, LdrColor/HdrColor, depth, normals, motion, semantic or instance products supported by the installed add-on/runtime.
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
    - aovs
---
# OVRTX render products and AOVs

Use the add-on's documented render-product controls to request the products the
installed runtime actually exposes. The OVRTX worker/native client is an
installed dependency; users need the add-on and its documented runtime,

## When to Use

Use for beauty, LdrColor/HdrColor, depth, normals, motion, semantic or instance products supported by the installed add-on/runtime.

## Instructions

1. Record the source `.blend` or USD, active camera, resolution/aspect, frame or
   simulation time, add-on/runtime identity, product paths, requested RenderVars,
   color mode, and output directory. Work in a copy and hash the source.
2. Enumerate products and variables through the add-on status/report surface.
   A product is not interchangeable with a Blender viewport screenshot. Keep
   sensor cameras, render-product cameras, and review camera projections
   explicit.
3. Request only named variables supported by the current runtime. Common
   products include `LdrColor` (display-encoded RGBA8) and `HdrColor`
   (scene-linear HDR); depth, normals, IDs, motion, and semantics are optional
   capabilities and must be reported as unavailable when absent.

## Capture native data

1. Start or reuse the documented warm session, create/select the render product,
   and wait for a completed sample. Do not treat a successful request as a
   frame; decode the returned buffer and check its dimensions, channel count,
   orientation, timestamp, and product/RenderVar identity.
2. Preserve native arrays in their original numeric format (including float
   depth, integer IDs, and HDR values). Write a per-variable checksum and a
   manifest containing source hash, camera matrix/intrinsics, frame/time,
   resolution, sample boundary, and runtime status.
3. If several sensors/products are requested, read each explicitly and require
   all expected paths. Fail on missing, stale, duplicate, or out-of-order data.
   Never fill missing pixels or infer AOVs from a beauty image.

## Produce review derivatives

- Keep raw/native data immutable. Make separately named PNG/EXR/NPY/JSON review
  derivatives with the conversion, normalization, colormap, and clipping range.
- Project depth/IDs/points only with the exact selected camera and scene
  transform. A depth-tested overlay or false-color map is review evidence, not
  the native product.
- Label display-transform ownership. For scene-linear HDR, convert once in the
  consumer; for OVRTX display-encoded LdrColor, do not apply Blender's view
  transform a second time.

## Acceptance bundle

Return native buffers, product/RenderVar paths, camera and timing metadata,
checksums, effective settings, logs, and review previews. Classify each output
as `native`, `converted-review`, `viewport`, `synthetic`, or `unavailable`.
Pair with `ovrtx-color-management` and `blender-image-evidence-review` for
display and visual checks.

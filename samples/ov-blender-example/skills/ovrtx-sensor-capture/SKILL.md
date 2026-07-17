---
name: ovrtx-sensor-capture
description: Capture OVRTX sensor and render-product outputs with camera alignment, provenance, and separated raw versus review products. Use for LiDAR, multi-sensor render variables, semantic/instance products, depth, normals, or other AOV requests supported by the installed add-on/runtime.
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
# OVRTX Sensor Capture

Capture OVRTX sensor and render-product outputs with camera alignment,
provenance, and clearly separated raw versus review products.

## When to Use

Use for LiDAR, multi-sensor render variables, semantic/instance products, depth, normals, or other AOV requests supported by the installed add-on/runtime.

## Boundary

- Use the add-on and supported runtime through their documented interfaces.
- Do not invent sensor returns, fill missing points, or call a Blender overlay a
  native sensor product.
- Keep acquisition, geometric visibility filtering, and display compositing as
  separate stages with separate labels.

## Instructions

1. Record source scene, camera matrix/intrinsics, sensor/render-product paths,
   output variables, frame/time, runtime/add-on identity, and configuration hash.
2. Validate the requested sensor profile and restart requirements before capture.
   A profile that changes immutable emitter-array shape must use an explicit new
   session rather than a silent live mutation.
3. Capture native arrays/products first. Preserve timestamps, sensor pose,
   return/instance identity, dimensions, orientation, and checksums.
4. For camera-visible views, project raw world points through the exact selected
   camera and apply an evaluated-geometry first-surface test before reducing or
   drawing points.
5. If accumulating point history, label it as visualization history. Retain
   genuine world-space returns; never turn it into synthetic sensor data.
6. Produce raw-only, depth-tested, and optional beauty-overlay outputs separately.
   For semantic/AOV products, retain native arrays/ID maps alongside colored
   review images.

## Acceptance bundle

Return native arrays or render variables, sensor/camera configuration, raw and
filtered counts, first/middle/last previews, requested images/movie,
projection/alignment report, checksums, and a classification of native,
reconciled, synthetic, viewport, or cache-only artifacts. Stop on missing or
unsupported output variables instead of substituting a screenshot.

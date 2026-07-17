---
name: ovrtx-semantic-aov-capture
description: Request and validate OVRTX semantic, instance, depth, normal, mask, and other render-product AOVs from Blender while preserving ID maps, camera alignment, output-variable provenance, and raw-versus-display distinctions. Use when training data, compositing, segmentation, or sensor evaluation needs more than an RGB beauty image.
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
# OVRTX semantic and AOV capture

Treat each AOV as a typed render product with explicit ownership and
provenance. A colorized screenshot, Blender viewport overlay, or postprocessed
mask is not a native semantic/instance product unless the report says so.

## When to Use

Use when training data, compositing, segmentation, or sensor evaluation needs more than an RGB beauty image.

## Runtime boundary

Use the add-on's documented render
product/output-variable controls and documented capability probes. If a requested
variable is unsupported, return `blocked` and preserve the exact capability
message.

## Instructions

1. Work on a copy and record source hash, Blender/add-on/runtime versions,
   camera matrix/intrinsics, resolution, frame/time, and color-management.
2. Assign stable object/prim identities before capture. For semantic labels,
   choose a documented namespace and record label token → prim/object mapping.
   For instance IDs, ensure each independently addressable instance has a
   stable ID across frames; do not derive IDs from selection order or pixel
   colors.
3. Verify visibility, collection exclusions, material/GeomSubset bindings, and
   evaluated geometry before requesting products. Record whether IDs come from
   authored USD primvars, add-on metadata, or runtime-generated identity.

## Request native products

Request only variables exposed by the installed add-on/runtime, such as RGB,
depth, normals, motion vectors, semantic IDs, instance IDs, masks, or custom
render variables. Retain the render-product path, variable name/type, channel
shape, numeric encoding, units/range, orientation, frame/timestamp, and
content checksum. Keep direct children/order constraints explicit when the
runtime requires them; do not assume every output variable can be reordered.

Capture native arrays/images before display conversion. Keep scene-linear/HDR,
integer ID maps, metric depth, and display-ready colorized previews in separate
files. Apply a color palette or normalization only to a derived review image;
never overwrite raw IDs or depth with that presentation.

## Validate geometry and camera

- Render a tiny labeled fixture first: two objects with different semantic
  labels and instance IDs, plus a depth-separated occluder.
- Confirm every expected label/instance appears in the raw product and that
  occlusion and depth ordering agree with the authored camera.
- Compare image dimensions, orientation, intrinsics, frame/time, and camera
  identity to the RGB product. Reject mirrored, transposed, stale-camera, or
  stale-stage output.
- For animated scenes, validate first/middle/last frames and ID stability;
  record births/deletions explicitly.
- Check integer IDs are lossless and depth units are documented. Do not infer
  semantics from RGB or use a beauty render to fill absent pixels.

## Evidence contract

Write `ovrtx-aov-report.json` containing source/runtime/camera hashes,
requested variables, product paths/types, shape/dtype/units, label and instance
maps, frame checks, raw checksums, derived preview paths, render class, and
`status: pass|blocked|fail`. Retain native products, a small labeled fixture,
RGB/depth/ID previews, and any colorization palette. Mark outputs as native,
reconciled, synthetic, viewport, or cache-only. A pass requires raw product
availability, expected IDs/variables, camera/frame alignment, and unchanged
source hashes.

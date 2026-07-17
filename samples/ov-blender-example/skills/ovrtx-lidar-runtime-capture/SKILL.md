---
name: ovrtx-lidar-runtime-capture
description: Capture and validate runtime LiDAR returns through the OVRTX Blender add-on, including sensor profiles, camera alignment, raw point arrays, depth-visible overlays, and optional multi-pose scans. Use when a user wants trustworthy OVRTX LiDAR data or a point-cloud view rather than a synthetic Blender overlay.
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
# OVRTX LiDAR runtime capture

Capture real OVRTX returns first, then derive any review image. The sensor
configuration, world-space points, camera projection, and visibility filtering
must remain inspectable and reproducible.

## When to Use

Use when a user wants trustworthy OVRTX LiDAR data or a point-cloud view rather than a synthetic Blender overlay.

## Boundary and source assumptions

The user needs Blender, the add-on, and a supported runtime installation. Use
the add-on panel and documented
probes when available. If the runtime does not expose a LiDAR profile or raw
array, report `blocked` rather than drawing synthetic points and calling them
native.

## Instructions

1. Copy the source scene and choose an output directory such as
   `.cache/ovrtx-lidar-YYYYMMDD-HHMMSS/`. Record source hash, Blender/add-on
   versions, runtime identity, GPU, frame/time, sensor path, profile, pose, and
   the exact authored camera transform/intrinsics and resolution.
2. Preflight the sensor profile. Record channel count, scan rate, range,
   return policy, coordinate convention, intensity/semantic fields, and whether
   changing the profile requires a new runtime session. Never silently mutate
   an immutable emitter-array shape in a warm session.
3. Start the add-on's native capture and retain raw arrays before filtering:
   point/return identity, timestamp, sensor pose, range, intensity and labels
   when provided, dimensions, orientation, and checksums. Preserve the raw
   values in a machine-readable artifact; a screenshot is not sensor data.
4. Transform returns into world space with the recorded sensor pose. Project
   them through the exact selected Blender camera. Validate one asymmetric
   isolated target first to catch axis/sign, handedness, time, or camera errors;
   never correct an alignment error with a screen-space offset.
5. For camera-visible products, test the points against evaluated scene
   geometry in depth order. Record raw, in-frame, first-surface-visible, and
   drawn counts. Apply screen-cell reduction/decimation only after the geometry
   test and label that result as a visualization product.
6. Generate separate raw-only point cloud, depth-visible overlay, and optional
   beauty composite. The beauty image may be Blender or OVRTX; label the render
   owner and keep native arrays beside it. For an animated or multi-yaw scan,
   hash each phase independently and verify pose/time changes before aggregate
   rendering.

## Quality checks

- The authored camera is unchanged; camera identity, matrix, intrinsics, and
  projection are present in the report.
- Every raw point and metadata value is finite and has a stable frame/return
  identity. Unexpected count or bounds jumps trigger a warmup/timing review.
- The isolated-target alignment passes before a complex-scene claim.
- Points labeled visible are consistent with evaluated first-surface depth;
  floor/wall returns must not mask requested hero geometry.
- World-space accumulation is labeled as history and contains only genuine
  returns. Never fill holes with random, interpolated, or mesh-sampled points.

## Evidence contract

Write `ovrtx-lidar-report.json` containing source/runtime/camera/profile hashes,
capture times, raw and filtered counts, bounds, per-phase checksums, alignment
error, output paths, and `status: pass|blocked|fail`. Retain raw arrays,
calibration/alignment data, first/middle/last previews, and any movie/contact
sheet under the run directory. If the runtime exposes only an image or no raw
return schema, state that limitation and classify the product as image-only or
diagnostic; do not upgrade it to a LiDAR proof.

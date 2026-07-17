---
name: ovrtx-render-sequence
description: Render an animated Blender scene through a warm OVRTX session as a contiguous, validated frame sequence or movie. Use for camera moves, animated transforms, OVPhysX replay, frame ranges, contact sheets, and delivery movies.
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
# OVRTX Render Sequence

Render animated Blender scenes through a warm OVRTX session as contiguous,
validated frame sequences or movies.

## When to Use

Use for camera moves, animated transforms, OVPhysX replay, frame ranges, contact sheets, and delivery movies.

## Boundary

- Use the installed add-on's documented render/clip path and OVRTX runtime
  or native client. Use only documented add-on and runtime interfaces.
- Keep source `.blend` and authored USD immutable; write generated frames and
  reports to a caller-provided output directory.
- Keep OVRTX output distinct from Blender/Cycles reference output and review
  overlays.

## Instructions

1. Record source/add-on/runtime identity, scene and camera, frame range,
   resolution, samples/refinement target, color management, output format, and
   active GPU.
2. Render one smoke frame plus first/middle/last frames. Verify the requested
   camera and scene state actually change before starting a long sequence.
3. Prefer one persistent session for adjacent frames. Apply each frame's scene
   update atomically, advance to the requested completed-sample boundary, then
   read back once. Do not reboot the worker for every frame.
4. Preserve temporal history unless a reset is requested. Make every reset or
   resume boundary explicit and never hide a shard discontinuity.
5. Validate dimensions, alpha/orientation, nonblank structure, frame identity,
   completed samples, and checksums. Encode a movie only after the image sequence
   passes.
6. Produce a contact sheet and inspect motion, framing, settling, flicker,
   lighting discontinuities, and final readability.

## Acceptance bundle

Return the image sequence or movie, first/middle/last frames, contact sheet,
`manifest.json` with source/runtime/camera/frame metadata, per-frame checksums,
timing and completed-sample records, and an explicit classification of native,
cached, or postprocessed outputs. If runtime sequence capture is unavailable,
report the limitation instead of silently falling back to unrelated renders.

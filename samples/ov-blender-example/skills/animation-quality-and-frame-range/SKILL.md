---
name: animation-quality-and-frame-range
description: Author, render, and review Blender animation over a declared frame range, including keyframes, shape keys, drivers, texture states, and camera motion. Use for OVRTX sequences, contact sheets, flicker/framing checks, or export decisions through the installed add-on/runtime.
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
    - animation
---
# Animation quality and frame range

An animation is accepted only after its timing, visual continuity, and delivery
format are checked. Keep authored Blender animation and authoritative OVPhysX
pose replay distinct.

## When to Use

Use for OVRTX sequences, contact sheets, flicker/framing checks, or export decisions through the installed add-on/runtime.

## Instructions

1. Save a copy and record scene/add-on/runtime identity, FPS, frame start/end,
   subframes, active camera, resolution, color presentation, and output format.
2. Set keyframes on named objects/properties. Use Bezier for organic motion,
   Linear for constant-speed mechanisms, and shape keys/drivers/NLA only when
   the target export/runtime supports them. Set explicit interpolation and
   avoid hidden Python frame handlers for portable delivery.
3. For texture states, register all images to one canvas and use per-layer
   holds, masks, opacity, emission, or UV changes. Do not crossfade unrelated
   full-frame crops or animate background/HUD pixels as part of an asset.
4. For OVPhysX motion, use `ovphysx-simulation-workflow` and replay complete
   authoritative pose samples; never replace simulation with hand-authored
   ballistic keyframes while claiming physics output.

## Render and review

1. Render one smoke frame plus first, middle, and last frames. Confirm the
   intended camera and subject change before a long capture.
2. Use `ovrtx-render-sequence` for a contiguous warm session when OVRTX output
   is requested. Apply frame updates atomically, wait for the requested sample
   boundary, and read back once. Make reset/resume boundaries explicit.
3. Validate frame dimensions, alpha/orientation, checksums, nonblank structure,
   completed samples, and timing. Keep native OVRTX frames separate from
   Blender/Cycles/EEVEE references.
4. Build a contact sheet and inspect subject dominance, silhouette stability,
   texture registration, flicker, lighting continuity, framing, settling, and
   final-frame readability. Reject and repair a failed dimension before export.

## Delivery gates

- A movie is encoded only after the image sequence passes review.
- State whether effects are export-compatible, image-sequence-only, or
  Blender-Python runtime behavior; do not claim GLB playback for unsupported
  handlers.
- Return frames/movie, contact sheet, frame manifest, per-frame checksums,
  camera/FPS/range metadata, and known limitations.
- If the runtime or sequence capability is unavailable, report the
  blocker instead of silently substituting another renderer.

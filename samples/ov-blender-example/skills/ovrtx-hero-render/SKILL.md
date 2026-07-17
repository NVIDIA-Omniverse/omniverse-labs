---
name: ovrtx-hero-render
description: Produce a polished, reproducible OVRTX still or short shot from a user Blender scene with deliberate camera, lighting, materials, samples, color management, and review gates. Use for portfolio images, product hero frames, or final presentation output.
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
# OVRTX hero render

This is a presentation workflow built on the current-scene add-on path. It
works for any user scene and uses the add-on's supported runtime setup.

## When to Use

Use for portfolio images, product hero frames, or final presentation output.

## Instructions

1. Duplicate/save the source `.blend`, hash it, and make a shot directory with
   camera, frame, resolution, aspect, render engine, samples, output format,
   and color-presentation mode.
2. Set one named camera explicitly. Check framing, clipping, focal length or
   orthographic scale, focus distance/DOF, horizon, and object readability.
3. Build a deliberate three-point or world-plus-key setup using
   `ovrtx-lighting-and-world`; keep a neutral control and avoid changing
   materials to compensate for exposure.
4. Validate materials/UVs and ensure the active camera sees the intended
   geometry. Hide collision/debug helpers from the beauty product.

## Render and refine

1. Run add-on preflight, render one low-sample smoke frame, and verify OVRTX
   engine identity, camera, dimensions, nonblank structure, and output path.
2. Raise samples/refinement only after the smoke passes. Keep temporal/session
   identity stable for a short animated shot and use one warm session for
   adjacent frames.
3. Request the chosen LdrColor or HdrColor mode and record display-transform
   ownership. Preserve the raw native image before any crop, grade, label, or
   sharpening.
4. For a final still, render at the requested dimensions and inspect highlights,
   shadows, edges, contact, DOF, and material response. For a short shot, render
   first/middle/last plus a contact sheet and check flicker and framing drift.

## Deliverable contract

Return the label-free native OVRTX image/movie, optional Blender/Cycles control,
contact sheet, and `manifest.json` with source/add-on/runtime identity, camera,
frame range, resolution, completed samples, color mode, settings, checksums,
timestamps, and status. Mark all crops, grades, overlays, and comparisons as
postprocessed review derivatives. Use `blender-image-evidence-review` before
calling the shot ready.

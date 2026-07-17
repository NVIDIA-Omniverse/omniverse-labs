---
name: blender-creative-evidence-runner
description: Run iterative, reviewable creative Blender and OVRTX scene work from a user-owned .blend, including camera, lookdev, animation, and presentation renders. Use when refining a shot or reporting creative and technical status separately.
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
    - validation
---
# Blender creative evidence runner

Use this skill for repeatable shot development, not product issue tracking. The add-on and installed OVRTX runtime are the execution boundary; Use only documented add-on and runtime interfaces.

## When to Use

Use when refining a shot or reporting creative and technical status separately.

## Instructions

1. Make a take directory and record source hash, add-on/runtime identity, camera, frame(s), resolution, color mode, and creative target.
2. Preserve each take. Record feedback, visible change, technical status (`blocked`, `partial`, `passing`), and creative status (`exploring`, `iterating`, `review-ready`, `approved`).
3. Run a low-sample smoke through the current-scene workflow. Confirm camera, dimensions, nonblank image, and engine/product identity before expensive renders.
4. Refine one variable group at a time (composition, lighting/world, material/UV, or animation). Keep native output separate from Blender controls and postprocessed overlays.
5. For motion, retain first/middle/last frames and a contact sheet; verify frame range, temporal continuity, and no framing drift.

## Review and handoff

Use image-evidence review for visible proof. A successful API call is insufficient without an image or validated report. Return the approved take, manifest, native images/movie, optional Blender/Cycles control, and known limitations. Do not close a creative iteration merely because a technical render passed.

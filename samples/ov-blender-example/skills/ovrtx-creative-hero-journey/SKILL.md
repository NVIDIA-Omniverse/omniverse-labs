---
name: ovrtx-creative-hero-journey
description: Turn a copied current Blender scene into a polished OVRTX hero still through a concise artist workflow covering preflight, camera and look refinement, a low-cost smoke render, final native output, review, and honest provenance. Use when an artist wants the shortest reliable path from scene to presentation-ready OVRTX image.
---

# OVRTX creative hero journey

Use this as the short artist orchestrator. Route detailed checks to the named public skills and preserve their acceptance gates.

## Happy path

1. Run `blender-content-safety-and-privacy`, then create a unique shot directory, save a working copy of the supplied `.blend`, hash the source and copy, and leave the source unchanged. Keep linked assets available to the copy and stop if required content cannot be handled safely.
2. Run `ovrtx-addon-install-and-preflight`. Record the public version and readiness results. Treat a missing renderer, runtime, GPU, or required capability as `blocked`; do not substitute another renderer while claiming OVRTX output.
3. Run `ovrtx-current-scene-workflow` to name the active camera, confirm framing and renderable content, and establish the current-scene OVRTX path. Preserve the user's composition unless changes are requested.
4. Refine only what the shot needs: use `ovrtx-materialx-openpbr` for assigned materials, `ovrtx-lighting-and-world` for lights and environment, and `ovrtx-color-management` for a single explicit display-transform owner. Keep a neutral control while tuning the look.
5. Run one low-resolution, low-sample OVRTX smoke frame. Confirm engine identity, camera, dimensions, visible structure, material and light response, color mode, and output path before increasing cost.
6. Run `ovrtx-hero-render` for the final dimensions and samples. Preserve the label-free native output before any crop, grade, sharpening, annotation, or comparison.
7. Run `blender-image-evidence-review` on the final and any derivatives. Deliver the native hero image, optional clearly labeled review derivatives, and a manifest with source hash, public component versions, camera, frame, dimensions, completed samples, color ownership, timestamps, output hashes, status, and limitations.

## Completion gate

Call the journey complete only when preflight and smoke pass, the final is confirmed as native OVRTX output, the image review finds no blocking defect, and the manifest distinguishes native output, controls, and postprocessed derivatives. Otherwise return `blocked` or `fail` without overstating provenance.

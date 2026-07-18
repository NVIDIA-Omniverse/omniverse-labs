---
name: reference-to-3d-reconstruction
description: Reconstruct a measured Blender asset from one or more image references, wireframes, logos, or orthographic views, then validate scale, registered cameras, landmarks, silhouette, materials, and OVRTX output. Use for reference-locked modeling, orthographic image registration, pixel-to-world calibration, or explicit inferred-depth reporting; it needs Blender/MCP and the installed add-on.
---

# Reference to 3D reconstruction

Treat supplied images and measurements as the source of truth. The goal is a
repeatable asset with documented assumptions, not a plausible but unmeasured
mesh.

## Plan and register references

1. Preserve originals and record file hashes, pixel dimensions, orientation,
   crop, lens/orthographic assumptions, and known dimensions. Do not silently
   stretch a reference to fit a camera.
2. Classify the task: single-view concept, front/side/top orthographic fit,
   multi-view reconstruction, wireframe-to-mesh, logo/mascot, or texture-driven
   surface. For single views, label depth as inferred.
3. Establish Blender units and a named `REF_*` collection. Add image empties or
   planes with locked transforms; keep them out of final renders/exports.
4. Create measurable landmarks (corners, centerlines, extrema, joints, and
   silhouette breaks). Store expected image-space positions and tolerances.

Before modeling, read [references/reference-registration.md](references/reference-registration.md).
Follow its idempotent image-empty recipe, camera conventions, calibration and
landmark schemas. Run `scripts/reference_projection_report.py` in Blender after
registration and after any camera, scale, or silhouette change. Do not accept a
visual overlay without the resulting projection report.

## Reconstruction loop

1. Block out with primitives at the measured scale. Lock the reference camera
   and solve projection before adding detail.
2. Fit the primary silhouette and major parts across all available views.
   Use separate objects for independently moving parts and preserve stable
   names. Avoid adding hidden geometry to cover a projection mismatch.
3. Refine only after silhouette/landmark error is within the declared
   tolerance. Apply controlled bevels, subdivision, and topology repair while
   checking proportions at thumbnail size and in greyscale.
4. Assign materials/UVs through `texture-uv-material-workflow`; validate the
   reference look separately from geometric fit.
5. Render a neutral Blender control and the same camera through
   `ovrtx-current-scene-workflow`. Compare framing, silhouette, normals,
   occlusion, and color with label-free images plus a numeric fit report.

Use `blender-community-skill-bootstrap` to install upstream `blender-modeling`
and `blender-cameras` recipes when those optional skills are absent. Use
`blender-python-execution` for the MCP transaction. Keep reference images in a
named `COL-reference` collection and create them through Blender's data API;
do not depend on the current selection.

Use `blender-camera-framing` only for beauty or diagnostic subject containment.
Do not use it to solve a registered reference camera: registration fixes the
camera from calibration and measures object error against that camera.

## Acceptance gates

- Reference files and source `.blend` remain immutable; derived outputs use a
  caller-selected cache/output folder.
- Every fit claim names camera/projection, landmarks or mask metric, tolerance,
  and view(s) tested. “Looks close” is not a fit result.
- Image planes, guides, and diagnostic overlays are excluded from beauty renders
  and USD unless explicitly requested.
- Mark unseen depth, occluded surfaces, and inferred materials as assumptions.
- Require every view to state whether depth is measured, constrained by another
  view, or inferred. A passing 2D projection report does not validate depth.
- A failed OVRTX render is a runtime/add-on diagnostic; use the documented
  status and log surfaces to diagnose it.

## Handoff

Return the requested `.blend` or export plus the camera/scale assumptions, fit
metrics, and unresolved ambiguities needed to judge the reconstruction. Add a
reference manifest, source map, or previews only when the handoff benefits from
them. For USD/SimReady, continue with the dedicated skill.

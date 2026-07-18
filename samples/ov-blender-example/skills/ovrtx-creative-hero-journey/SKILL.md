---
name: ovrtx-creative-hero-journey
description: "Take a current Blender scene through a concise artist-facing path to a polished native OVRTX hero still: preserve a derivative, preflight, refine camera/look, run a smoke, render the final, and label provenance honestly. Use when an artist wants the whole presentation workflow without manually selecting every specialist skill."
---

# OVRTX creative hero journey

This is a generic user-outcome orchestrator, not an evidence-production or
deployment workflow.

1. Preserve the source and work on a caller-owned derivative.
2. Run `ovrtx-addon-install-and-preflight`; stop on a missing runtime, native
   client, GPU, or engine rather than substituting another renderer.
3. Run `ovrtx-current-scene-workflow` and `blender-camera-framing` for the named
   shot. Preserve composition unless the user requested a change.
4. Refine only needed surfaces with `ovrtx-materialx-openpbr`,
   `texture-uv-material-workflow`, and `ovrtx-lighting-and-world`.
5. Choose explicit samples and display ownership through
   `ovrtx-render-settings` and `ovrtx-color-management`.
6. Run `ovrtx-hero-render`: first a one-sample smoke, then the requested native
   final to a distinct output directory.
7. Inspect the final image. Keep Blender controls and postprocessed derivatives
   clearly separate from native OVRTX output.

Complete the journey when preflight, camera containment, smoke, and native final
all pass. Summarize the image and any remaining limitation. Create a manifest,
review sheet, checksums, or comparison package only when requested.

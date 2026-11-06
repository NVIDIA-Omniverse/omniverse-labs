---
name: ovrtx-color-management
description: Choose and validate ownership of OVRTX display presentation between scene-linear HDR and display-encoded LDR, using the installed add-on enum and the official color-presentation probe. Use for double-transform, washed-out, too-dark, or Blender-versus-OVRTX display differences.
---

# OVRTX color management

Keep exactly one display-transform owner:

- `scene_linear_hdr`: OVRTX supplies scene-linear data and the consumer applies
  Blender/OCIO view settings once.
- `ldr_rgba8_display_passthrough`: OVRTX supplies display-encoded pixels and the
  consumer must not apply Blender's display transform again.

Probe the allowed `color_presentation_mode` enum through
`ovrtx-current-scene-workflow`, then set it with `ovrtx-render-settings`.
Do not assume these identifiers exist in another add-on revision.

For the root-owned Windows port, run the focused ownership controls from the
repository root:

```text
pixi run python -m pytest source/ovrtx-addon/tests/test_color_presentation.py source/ovrtx-addon/tests/test_viewport_display_overlay.py -q
```

The worker does not inherit Blender's `OCIO`; the Blender display configuration
remains unchanged. LDR passthrough and HDR correction must be tested separately.
For a real scene, render the same camera/light/material state with both the
requested native mode and a clearly labeled Blender control. Diagnose runtime
readback and dimensions first, then camera/stage, then light/world, then color
ownership, and only then material values. Never fix a double transform by
editing albedo, light energy, or exposure invisibly.

Summarize the chosen owner, enum, native result, and first mismatch. Generate
comparison sheets or numeric image reports only when requested.

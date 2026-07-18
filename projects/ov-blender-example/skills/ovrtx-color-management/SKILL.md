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

For the pinned official example, run the public ownership regression:

```text
python3 scripts/run_ovrtx_color_presentation_probe.py \
  --output-dir /absolute/caller-owned/color-probe
```

For a real scene, render the same camera/light/material state with both the
requested native mode and a clearly labeled Blender control. Diagnose runtime
readback and dimensions first, then camera/stage, then light/world, then color
ownership, and only then material values. Never fix a double transform by
editing albedo, light energy, or exposure invisibly.

Summarize the chosen owner, enum, native result, and first mismatch. Generate
comparison sheets or numeric image reports only when requested.

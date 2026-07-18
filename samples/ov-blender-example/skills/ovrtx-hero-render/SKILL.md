---
name: ovrtx-hero-render
description: Produce a polished native OVRTX still from a user Blender scene with explicit camera framing, materials, lighting, sample bounds, color ownership, and a low-cost smoke through the registered OVRTX Blender engine. Use for product, portfolio, or presentation hero frames; do not substitute another Blender engine while claiming OVRTX output.
---

# OVRTX hero render

## Prepare the shot

1. Work on a caller-owned scene derivative.
2. Run `ovrtx-addon-install-and-preflight` and
   `ovrtx-current-scene-workflow`.
3. Fit and independently verify the named camera with
   `blender-camera-framing`.
4. Use `ovrtx-materialx-openpbr`, `texture-uv-material-workflow`, and
   `ovrtx-lighting-and-world` only where the shot needs them.
5. Configure sample bounds and presentation ownership with
   `ovrtx-render-settings` and `ovrtx-color-management`.

## Render through the registered OVRTX engine

After `ovrtx-render-settings` has selected `OVRTX_EXAMPLE`, run one small smoke
through the bounded helper from `blender-render-and-export`:

```text
blender --background /absolute/caller-owned/hero.blend \
  --python skills/blender-render-and-export/scripts/render_or_export.py -- \
  --operation render_still \
  --output /absolute/caller-owned/hero-smoke.png \
  --camera CAM-hero --frame 1 --resolution 640 360 100
```

Require `ok: true`, `detail.engine: OVRTX_EXAMPLE`, the requested camera and
dimensions, a nonzero file, and a finite nonblank image. Then rerun to a new
output path with the caller's final dimensions/sample bound. Never overwrite
the raw native frame; crop, grade, sharpen, or annotate only a clearly labeled
derivative.

## Finish

Inspect framing, clipping, highlights, shadow detail, contact, material
response, and color ownership. Return the native image and a concise pass or
blocker. Add controls, hashes, manifests, or comparison images only when the
caller needs reproducibility or review.

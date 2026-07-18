---
name: blender-render-and-export
description: Render Blender stills and export named scene objects to GLB or USD with explicit cameras, frames, absolute output paths, context preservation, operator capability checks, file verification, and structured JSON results. Use when an agent must produce a generic Blender render or interchange file without relying on optional community skills or confusing Blender output with native OVRTX output.
---

# Blender render and export

Use `scripts/render_or_export.py` for the fragile final write. It supports one
operation per invocation: `render_still`, `export_glb`, or `export_usd`.
Read `references/render-export-contract.md` before choosing settings.

For MCP, prepend `RENDER_EXPORT_REQUEST` and append the entire helper source:

```python
RENDER_EXPORT_REQUEST = {
    "operation": "render_still",
    "output_path": "/absolute/caller-owned/preview.png",
    "camera_name": "CAM-product",
    "frame": 1,
    "resolution": [960, 540, 100],
}
```

For a saved derivative:

```text
blender --background scene.blend --python scripts/render_or_export.py -- \
  --operation export_glb --output /absolute/caller-owned/asset.glb \
  --objects GEO-body GEO-trim
```

To save the authored scene itself, use the bounded save-copy recipe in the
contract. This is separate from render/export so a failed derivative write does
not unexpectedly rename the working file.

The helper refuses relative paths, missing cameras/objects, incompatible file
extensions, unavailable operators, empty output, and output outside a
caller-owned destination. It restores camera/frame/render-path settings after a
still and preserves object selection and active object for exports. It does not
save or overwrite the `.blend`.

## Acceptance

- Require `ok: true`, the requested operation, an absolute output path, a
  nonzero byte count, and SHA-256.
- For a still, require the requested camera/frame/effective resolution and
  inspect the current image for cropping, blank output, materials, and lighting.
- For GLB/USD, run `scripts/inspect_roundtrip.py` in a fresh disposable
  background Blender process. It resets that process, imports one derivative,
  and reports object names, finite bounds, materials, units, and actions.
- Label Blender renders and Blender-exported files as such. Do not claim they
  are OVRTX render products or add-on-authored USD unless that separate workflow
  establishes it.

Rendering and export are explicit writes. Never invent an output directory,
silently overwrite an existing file, or render a full sequence when the user
requested only a still or smoke frame.

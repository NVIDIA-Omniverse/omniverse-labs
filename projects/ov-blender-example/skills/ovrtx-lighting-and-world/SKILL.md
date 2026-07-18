---
name: ovrtx-lighting-and-world
description: Author, inspect, and debug Blender 5.x lights and World/HDRI state for OVRTX conversion. Use when a scene is dark, overexposed, flat, contains unsupported light fields, or differs from a Blender reference.
---

# OVRTX lighting and world

Author lights and World state in Blender. OVRTX consumes that state through the
add-on's documented conversion and update paths. Blender inspection can prove
the authored inputs are coherent; only an identified native OVRTX render can
prove that the runtime converted and applied them.

## Author controlled inputs

1. Inspect the active scene, camera, World, exposure, lights, emissive meshes,
   and object scale before changing anything. Use stable names and preserve the
   user's existing lighting unless replacement was requested.
2. Establish one simple supported source first: a POINT, SPOT, SUN, or AREA
   light, or a World Background. Make a cheap render before adding complexity.
3. Set light type, energy, color or temperature, size/shape, transform, shadow
   intent, and spot/sun parameters explicitly. Aim lights from local `-Z`;
   Euler guesses are not a reliable targeting method.
4. Use either a constant World Background or a reachable Environment Texture
   graph. Resolve the HDRI path through Blender and set image color space
   intentionally. Do not silently replace a missing HDRI with a flat color.

Read [references/blender-5-lighting-recipes.md](references/blender-5-lighting-recipes.md)
before scripting lights or World nodes. It contains idempotent Blender 5.x
`bpy` transactions for named lights, target orientation, constant World
lighting, HDRI nodes, and safe state restoration.

## Audit Blender state

Run the read-only [scripts/audit_lighting_world.py](scripts/audit_lighting_world.py)
after authoring. It checks named or scene-visible lights, finite transforms and
colors, type-specific ranges, World output reachability, Background and
Environment Texture inputs, missing image files, and whether the scene has an
effective light source.

For MCP, prepend a request and append the complete script in one Python call:

```python
LIGHTING_AUDIT_REQUEST = {
    "scene": "Scene",
    "lights": ["Key", "Fill"],
    "require_effective_lighting": True,
}
# Append the complete contents of scripts/audit_lighting_world.py here.
```

For a caller-owned saved scene:

```bash
blender --background scene.blend \
  --python scripts/audit_lighting_world.py -- \
  --scene Scene --lights Key Fill --require-effective-lighting
```

The command prints `blender_lighting_world_audit.v1` JSON and exits `2` on a
failed check. Omitting `--lights` audits all non-hidden light objects in the
scene. Use `--allow-unsupported-types` only when the installed add-on documents
additional conversion support.

## Verify OVRTX conversion

Use the add-on's documented scene-refresh path after light type, node topology,
or environment-image changes. Supported value-only edits may use its warm
session update path. Then render the smallest completed native frame and verify
engine/product identity before comparing pixels.

Diagnose in this order: native render/readback → camera and selected product →
light transforms and scale → World/dome presence → light family and units →
exposure/display transform → materials. A Cycles, EEVEE, or viewport image is a
useful control, but it does not establish OVRTX conversion.

Summarize the changed Blender fields, audit result, unsupported inputs, and
native-render result. Create comparison images, logs, or a separate report only
when the user requests reproducibility, troubleshooting, or review artifacts.
When display-transform ownership affects the comparison, use
`ovrtx-color-management`, inspect the exact scene view settings, and label which
runtime produced each image.

---
name: animation-quality-and-frame-range
description: Author, render, and review Blender animation over a declared frame range, including keyframes, shape keys, drivers, texture states, and camera motion. Use for OVRTX sequences, contact sheets, flicker/framing checks, or export decisions through the installed add-on/runtime.
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
    - animation
---
# Animation quality and frame range

An animation is accepted only after its timing, visual continuity, and delivery
format are checked. Keep authored Blender animation and authoritative OVPhysX
pose replay distinct.

## When to Use

Use for OVRTX sequences, contact sheets, flicker/framing checks, or export decisions through the installed add-on/runtime.

## Instructions

1. Save a copy and record scene/add-on/runtime identity, FPS, frame start/end,
   subframes, active camera, resolution, color presentation, and output format.
2. Set keyframes on named objects/properties. Use Bezier for organic motion,
   Linear for constant-speed mechanisms, and shape keys/drivers/NLA only when
   the target export/runtime supports them. Set explicit interpolation and
   avoid hidden Python frame handlers for portable delivery.
3. For texture states, register all images to one canvas and use per-layer
   holds, masks, opacity, emission, or UV changes. Do not crossfade unrelated
   full-frame crops or animate background/HUD pixels as part of an asset.
4. For the supported native drop/contact control, use
   `ovphysx-drop-contact-acceptance` and replay complete authoritative pose
   samples. Other physics sequences need a probed official client or an add-on
   extension; never replace simulation with hand-authored ballistic keyframes
   while claiming physics output.

Use `blender-community-skill-bootstrap` to install upstream `blender-animation`
when detailed transform, constraint, shape-key, driver, or NLA recipes are
needed. Use `blender-python-execution` for bounded authoring calls; render first,
middle, and last smoke frames before starting a full sequence.

Read [references/blender-5-animation-api.md](references/blender-5-animation-api.md)
before authoring keyframes. It gives context-free Blender 5.x transactions for
object properties, interpolation, shape keys, drivers, and NLA-safe inspection.
Keep creation and inspection in separate MCP calls so a failed audit cannot
partially rewrite the animation.

After authoring, run the read-only `scripts/audit_animation.py` against explicit
object names and RNA property paths. For MCP, prepend a request and append the
complete script in the same Python execution:

```python
ANIMATION_AUDIT_REQUEST = {
    "objects": ["GEO-subject"],
    "properties": ["location", "rotation_euler"],
    "frames": [1, 24, 48],
    "require_keyframes": True,
    "require_motion": True,
}
# Append scripts/audit_animation.py here.
```

For a saved caller-owned scene, use:

```text
blender --background scene.blend --python scripts/audit_animation.py -- \
  --objects GEO-subject --properties location rotation_euler \
  --frames 1 24 48 --require-keyframes --require-motion
```

Require `status: pass`, no missing objects or properties, finite sampled values,
and the requested keyframe/motion checks. With `require_keyframes`, every named
property must have a direct key in the sampled interval; NLA, drivers, parents,
and constraints can still establish sampled motion when that gate is off. The
audit restores the original frame and never saves. A changing evaluated value
proves sampled motion, not that a
particular exporter or runtime can reproduce its source mechanism.

When the camera must contain an animated subject, run
`blender-camera-framing` at every declared framing sample (at minimum first,
middle, last, and motion extrema) and retain each safe-UV report. A fit at one
frame does not establish sequence-wide containment; choose whether the camera
is fixed to the union of sampled bounds or deliberately animated between fits.

## Render and review

1. Render one smoke frame plus first, middle, and last frames. Confirm the
   intended camera and subject change before a long capture.
2. For contiguous OVRTX capture, use `ovrtx-render-sequence` to probe and
   validate the developer contract. If the capability is absent, it routes to
   `extend-ovrtx-render-sequences`; do not emulate native output with a Blender
   frame loop.
3. Validate frame dimensions, alpha/orientation, nonblank structure, completed
   samples, and timing. Keep native OVRTX frames separate from
   Blender/Cycles/EEVEE references.
4. Inspect representative frames for subject dominance, silhouette stability,
   texture registration, flicker, lighting continuity, framing, settling, and
   final-frame readability. Reject and repair a failed dimension before export.

## Delivery gates

- A movie is encoded only after the image sequence passes review.
- State whether effects are export-compatible, image-sequence-only, or
  Blender-Python runtime behavior; do not claim GLB playback for unsupported
  handlers.
- Return the requested frames or movie with camera/FPS/range metadata and known
  limitations. Generate contact sheets, checksums, or a frame manifest only
  when the caller requests review or reproducibility.
- If the runtime or sequence capability is unavailable, report the
  blocker instead of silently substituting another renderer.

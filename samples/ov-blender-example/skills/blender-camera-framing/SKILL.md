---
name: blender-camera-framing
description: Fit a named set of evaluated Blender objects into a perspective or orthographic render camera with correct render aspect, lens shift, margin, target, clipping, and JSON verification. Use when an MCP or background-Blender task must frame an object reliably rather than merely frame an interactive 3D viewport.
---

# Blender camera framing

Use `scripts/camera_framing.py` when a named subject must be wholly visible in
the scene camera. The helper uses evaluated world bounds, the active render
dimensions and pixel aspect, and Blender's own camera frame. It does not guess
from an object's dimensions or from one generic field-of-view angle.

Read `references/framing-contract.md` before changing the helper or composing a
custom framing transaction.

## Choose the correct kind of framing

- **Render-camera framing** changes a Camera object and works in interactive or
  background Blender. Use this skill for a render, export, shot, or reproducible
  camera result.
- **Viewport framing** changes `RegionView3D` in one visible `VIEW_3D` area.
  `bpy.ops.view3d.view_selected` requires an area/region override, does not work
  in background mode, and does not fit the scene camera. Use it only to prepare
  an interactive screenshot and restore selection afterward.

Never report viewport framing as proof that the render camera sees the subject.

## Inspect before fitting

1. Name the Camera object and every subject root explicitly. Decide whether
   descendants and evaluated instances belong to the subject.
2. Inspect the current frame, render resolution percentage, pixel aspect,
   camera type, lens or orthographic scale, sensor fit, shift, constraints, and
   clipping.
3. Choose a per-edge image margin. `0.08` means every evaluated bound point must
   land inside UV `[0.08, 0.92]` on both axes.
4. Preserve the current viewing direction unless the user supplied a view.
   `view_from` is the world direction from target toward camera, not camera
   toward target.
5. Use the evaluated bounds center as target by default. An optional existing
   Empty can provide a deliberate off-center look target. Creating or moving an
   Empty is an explicit behavior, never an implicit side effect.

## Execute through Blender MCP

Use the provider's Python-execution tool. Prepend `MCP_CONFIG`, then append the
complete helper source in the same call:

```python
MCP_CONFIG = {
    "camera_name": "CAM-product",
    "object_names": ["GEO-product"],
    "margin": 0.08,
    "include_descendants": True,
    "include_instances": True,
    "set_scene_camera": True,
}
# Append the complete contents of scripts/camera_framing.py here.
```

The helper detects `MCP_CONFIG`, performs one bounded transaction, and prints
one `blender.camera-frame/v1` JSON object. Do not send only the path: the Blender
MCP process may not share the agent's filesystem. Do not add sleeps, viewport
operators, a render, or a save to this transaction.

For caller-owned offline work, run Blender directly:

```text
blender --background scene.blend --python scripts/camera_framing.py -- \
  --camera CAM-product --objects GEO-product GEO-label --margin 0.08 \
  --include-descendants --include-instances --set-scene-camera
```

Add `--save-as /absolute/caller-owned/output.blend` only when the user requested
that derivative. The helper never overwrites the input file implicitly.

## Validate the result

Require all of the following in the returned JSON:

- `ok` and `verification.fits` are true;
- no requested name is missing and at least one evaluated bound was used;
- `verification.outside_count` and `behind_count` are zero;
- projected U/V extrema remain inside the declared safe UV rectangle;
- fitted near/far clips enclose the reported camera-depth range;
- the camera type, effective render dimensions, pixel aspect, target, and
  evaluated object/instance counts match the request.

Then render one inexpensive representative image and inspect it. Bounds fitting
proves geometric inclusion, not a pleasing composition, unobstructed subject,
correct depth of field, or adequate negative space. Refine view direction,
target, lens, and margin deliberately; rerun the helper after each optics or
resolution change.

As an independent final check, run `blender-python-execution`'s
`scripts/scene_audit.py` against the same roots, margin, descendant policy, and
instance policy. Deliberately distinct fitting and audit implementations reduce
the chance that one algebra or selection bug certifies itself.

## Failure rules

- Missing objects, an empty-only subject, non-finite bounds, a zero view vector,
  a missing requested target Empty, unsupported camera projection, or explicit
  clips that exclude the subject are hard failures.
- Transform-affecting camera constraints are rejected by default. Remove or
  deliberately manage them in a separate transaction; do not fight their
  evaluated result silently.
- If camera shift plus margin places the optical axis outside the safe frame,
  choose a smaller shift/margin or a different explicit composition.
- If Geometry Nodes or collection instances are important, inspect the reported
  instance count and render a control. Instance discovery is dependency-graph
  best effort and is reported separately from direct evaluated objects.
- Refit at each required animation sample when the subject, target, camera
  parent, modifier output, or render settings animate.

# Camera framing contract

## Create or resolve the camera

Use the data API so camera creation does not depend on selection or an editor:

```python
import bpy

name = "CAM-requested"
camera = bpy.data.objects.get(name)
if camera is not None and (camera.type != "CAMERA" or camera.library is not None):
    raise RuntimeError(f"refuse incompatible camera target: {name}")
if camera is None:
    data = bpy.data.cameras.new(name + "-data")
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
camera.data.type = "PERSP"  # or ORTHO
bpy.context.scene.camera = camera
bpy.context.view_layer.update()
```

Do not silently reuse an unowned camera whose lens, constraints, animation, or
parenting must be preserved. Select a new stable name for an independent shot.

## Coordinate and margin definitions

Blender cameras look along local `-Z` with local `Y` up. `view_from` is a
normalized world-space direction from the target toward the camera. With the
camera rotation fixed, each bound point relative to the target is expressed in
camera-oriented coordinates `(x, y, z)`, and the camera is placed at local
`(0, 0, distance)`.

`margin` is the fraction reserved at each image edge. A margin `m` defines the
safe normalized image rectangle `[m, 1-m] x [m, 1-m]`; valid values satisfy
`0 <= m < 0.5`.

Use `camera.data.view_frame(scene=scene)`. In Blender 5.1 this frame incorporates
render resolution and percentage, pixel aspect, sensor fit, focal length or
orthographic scale, and `shift_x`/`shift_y`. Do not replace it with
`camera.data.angle`: one angle cannot describe both image axes and lens shift.

## Perspective fit

For every local camera-frame corner, divide X and Y by `-Z` and take the extrema
`left`, `right`, `bottom`, and `top`. Shrink them by the per-edge margin:

```text
safe_left   = left   + margin * (right - left)
safe_right  = right  - margin * (right - left)
safe_bottom = bottom + margin * (top - bottom)
safe_top    = top    - margin * (top - bottom)
```

The fixed look axis must lie inside these intervals. For point `(x, y, z)`, the
minimum camera distance is bounded by:

```text
z + x / safe_right
z + x / safe_left
z + y / safe_top
z + y / safe_bottom
z + epsilon
```

The maximum over every bound point and every expression is the exact fit for
the evaluated bounding corners under the selected view. This handles wide,
tall, asymmetric, shifted, portrait, and non-square-pixel frames.

## Orthographic fit

Evaluate `view_frame` temporarily at `ortho_scale = 1`. After applying the same
safe-edge interpolation, the required scale is the maximum of:

```text
x / safe_right, x / safe_left, y / safe_top, y / safe_bottom
```

over all points. Orthographic distance does not change projected size. Choose a
positive deterministic distance beyond the nearest point, then fit clips from
the resulting camera depths.

## Evaluated bounds

Resolve requested object names exactly. Optionally add their recursive children.
For direct objects, use the evaluated dependency-graph object, its evaluated
`bound_box`, and its evaluated `matrix_world`. This captures modifier and
Geometry Nodes output represented by the evaluated bound.

For dependency-graph instances, include an instance when its original object or
its instancing parent belongs to the resolved subject set. Transform its
evaluated bound corners by `DepsgraphObjectInstance.matrix_world`. Report direct
and instanced bound counts separately. Do not use empties, cameras, lights, or
speakers as geometry; an Empty may still be the explicit look target.

Keep all transformed corners for fitting. A merged world AABB is useful report
metadata but using only its eight corners can add unnecessary view-dependent
padding.

## Clipping and independent verification

After placement, transform all world bound points through the final camera
matrix. Their positive camera depths are `-local_z`. With padding `p`, fit:

```text
clip_start = max(minimum_clip, minimum_depth * (1 - p))
clip_end   = max(clip_start * 1.01, maximum_depth * (1 + p))
```

Explicit clips must already enclose the subject or the transaction fails.
Preserved clips produce a warning when they exclude it.

Finally project every bound point with
`bpy_extras.object_utils.world_to_camera_view`. Require finite coordinates,
positive depth, and U/V inside the safe rectangle within numerical tolerance.
This verification is independent of the fitting algebra and is the acceptance
gate.

## Viewport boundary

This contract owns a render Camera object. It does not inspect a Screen, select
objects, invoke `view3d.view_selected`, or alter a `RegionView3D`. Interactive
viewport framing must locate a real `VIEW_3D` area and `WINDOW` region, use
`bpy.context.temp_override`, and restore active/selected objects. It is a
different operation and cannot run in background Blender.

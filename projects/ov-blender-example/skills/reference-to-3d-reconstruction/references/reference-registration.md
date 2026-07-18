# Reference registration

Use this contract before changing geometry. Keep the source image immutable and
make camera, scale, and inference decisions inspectable.

## Coordinate and camera contract

- Use metres internally unless the caller specifies another Blender unit.
- A Blender camera looks along local `-Z`; local `+Y` is image up.
- Use an orthographic camera for an orthographic drawing. Do not compensate for
  drawing mismatch with perspective or non-uniform object scale.
- Record the screen-axis mapping for each view. Recommended mappings are front
  `horizontal=+X, vertical=+Z, depth=+Y`; side `horizontal=+Y or -Y,
  vertical=+Z, depth=+X`; top `horizontal=+X, vertical=+Y, depth=+Z`.
- Match the camera render aspect to the reference pixel aspect. For square
  pixels, set resolution to the image width and height and pixel aspect to 1.
  Set `Camera.sensor_fit = 'VERTICAL'` for this contract; then
  `Camera.ortho_scale` is the vertical world span and the horizontal span is
  the vertical span multiplied by render aspect. If preserving another sensor
  fit, calibrate it numerically with `world_to_camera_view` instead of assuming
  this relation.
- Preserve diagnostic cameras. Create a separate beauty camera.

## One-known-distance calibration

Choose two unambiguous pixels in the same orthographic view whose real distance
is known. With pixel points `a` and `b` and known world distance `d`:

```text
pixel_distance = hypot(b.x - a.x, b.y - a.y)
world_per_pixel = d / pixel_distance
image_width_world = image_width_px * world_per_pixel
image_height_world = image_height_px * world_per_pixel
camera.ortho_scale = image_height_world
```

Reject zero pixel distance, non-positive world distance, and a dimension that
is foreshortened or crosses unknown depth. Record the calibration in the
manifest rather than baking an unexplained Empty scale:

```json
{
  "known_distance": {
    "landmark_a": "left_extent",
    "landmark_b": "right_extent",
    "distance_world": 2.0,
    "unit": "m"
  }
}
```

## Locked image Empty

Create or update only a caller-owned named Empty. Never delete or rename an
existing unrelated object. Put all references in one collection and exclude
that collection from rendering and export.

```python
import bpy
from pathlib import Path

def ensure_reference_empty(image_path, object_name, camera, center,
                           width_world, collection_name="COL-reference"):
    path = str(Path(image_path).expanduser().resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(path)

    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    collection.hide_render = True

    obj = bpy.data.objects.get(object_name)
    if obj is not None and (obj.type != 'EMPTY' or obj.empty_display_type != 'IMAGE'):
        raise RuntimeError(f"refuse to replace non-image object: {object_name}")
    if obj is None:
        obj = bpy.data.objects.new(object_name, None)
        collection.objects.link(obj)
    elif obj.name not in collection.objects:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)

    image = bpy.data.images.load(path, check_existing=True)
    obj.empty_display_type = 'IMAGE'
    obj.data = image
    obj.empty_display_size = float(width_world)
    obj.location = center
    obj.rotation_euler = camera.rotation_euler
    obj.color[3] = 0.45
    obj.empty_image_depth = 'BACK'
    obj.hide_render = True
    obj.hide_select = True
    obj.lock_location = (True, True, True)
    obj.lock_rotation = (True, True, True)
    obj.lock_scale = (True, True, True)
    return obj
```

Set the Empty width from calibration, not by eye. Unlock through code only for
an intentional re-registration. Image empties are viewport guides; keep them
out of beauty renders, USD, and GLB.

## Landmark manifest

Pixel coordinates use top-left origin. A landmark may provide a literal world
point or an object-local point. Keep stable IDs across iterations.

```json
{
  "schema_version": 1,
  "thresholds": {
    "max_error_px": 4.0,
    "rmse_error_px": 3.0,
    "max_normalized_error": 0.01
  },
  "views": [
    {
      "id": "front",
      "image_path": "/absolute/caller-owned/reference.png",
      "image_size_px": [1000, 800],
      "camera": "CAM-reference-front",
      "projection": "ORTHO",
      "screen_axes": {"horizontal": "+X", "vertical": "+Z", "depth": "+Y"},
      "depth_status": "inferred",
      "depth_notes": "Only the front drawing constrains this view.",
      "known_distance": {
        "landmark_a": "left_extent",
        "landmark_b": "right_extent",
        "distance_world": 2.0,
        "unit": "m"
      },
      "landmarks": [
        {"id": "left_extent", "expected_px": [100, 400], "world": [-1, 0, 0]},
        {"id": "right_extent", "expected_px": [900, 400], "object": "GEO-subject", "local": [1, 0, 0]}
      ]
    }
  ]
}
```

Allowed `depth_status` values are `measured`, `constrained_by_other_views`, and
`inferred`. Explain the source or inference in `depth_notes`. The default
thresholds above are starting points, not universal quality claims; override
them in the manifest or on the report command line for the reference quality
and delivery requirement.

## Projection report

Run inside the registered `.blend`:

```bash
blender --background registered.blend \
  --python scripts/reference_projection_report.py -- \
  --manifest /absolute/reference-manifest.json \
  --output /absolute/reference-projection-report.json
```

The report uses `world_to_camera_view`, converts Blender bottom-left normalized
coordinates to top-left pixels, checks positive camera depth, and reports
per-landmark pixel and normalized error. A pass proves only the declared 2D
landmarks under the named camera. It does not prove silhouette fit, hidden
geometry, lens provenance, or inferred depth.

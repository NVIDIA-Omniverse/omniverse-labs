# Runnable library examples

Run examples from this directory using the main installed runtime Python, except
where indicated. Read each file for its arguments and output locations.

| Example | What it exercises | Environment |
| --- | --- | --- |
| physics_drop.py | Native rigid-body motion | libraries-env |
| render_materials.py | RTX material rendering | libraries-env |
| lidar_points.py | Native LiDAR coordinates and front/rear fisheye images at one scene revision | libraries-env |
| shared_stage.py | Shared scene updates | libraries-env |
| storage_revisions.py | Local file revisions | libraries-env |
| stream_roundtrip.py | Native local frame transfer and return command | libraries-env |
| operator_ui.py | Standalone UI controls sharing one value model | libraries-env |
| gantry_newton.py | Articulated gantry motion | newton-env |
| asset_package.py | Package creation and file verification | package-env |

For example, use `RUNTIME/libraries-env/Scripts/python.exe physics_drop.py` on
Windows. Set the runtime and output options documented in each example. The
application's **i → Example code** opens these same sources.
